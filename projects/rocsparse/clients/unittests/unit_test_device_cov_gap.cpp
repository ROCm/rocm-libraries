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

// Host paths the public clients never execute. Coverage counts a line only
// when every branch of that line has run, so each check below is called on
// both sides. Nothing here launches a kernel: the ILU0 x-impls return from
// argument checks, from the m == 0 / nnz == 0 shortcuts, or from the
// async-inplace dispatch (that case returns internal_error before the driver).

#include "unit_test_utils.hpp"

#include "rocsparse-debugging.h"
#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_templates.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace rocsparse_ut;

extern "C" rocsparse_status
    rocsparse_create_csr_descr_SWDEV_453599(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

namespace
{
    rocsparse_status null_csrsv(rocsparse_handle handle, size_t* buffer_size)
    {
        return rocsparse_csrsv_buffer_size<float>(handle,
                                                  rocsparse_operation_none,
                                                  0,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  buffer_size);
    }

    template <typename T>
    rocsparse_status null_bsrmv(rocsparse_handle handle)
    {
        return rocsparse_bsrmv<T>(handle,
                                  rocsparse_direction_row,
                                  rocsparse_operation_none,
                                  0,
                                  0,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr);
    }

    template <typename T>
    rocsparse_status null_gebsrmv(rocsparse_handle handle)
    {
        return rocsparse_gebsrmv<T>(handle,
                                    rocsparse_direction_row,
                                    rocsparse_operation_none,
                                    0,
                                    0,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr);
    }

    template <typename T>
    rocsparse_status null_csrmm(rocsparse_handle handle)
    {
        return rocsparse_csrmm<T>(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  0,
                                  0,
                                  0,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  0);
    }

    // csrsv_buffer_size is only specialized through a macro, so each precision
    // is a separate function rather than a template.
    rocsparse_status null_csrsv_double(rocsparse_handle handle, size_t* buffer_size)
    {
        return rocsparse_csrsv_buffer_size<double>(handle,
                                                   rocsparse_operation_none,
                                                   0,
                                                   0,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   buffer_size);
    }

    rocsparse_status null_csrsv_float_complex(rocsparse_handle handle, size_t* buffer_size)
    {
        return rocsparse_csrsv_buffer_size<rocsparse_float_complex>(handle,
                                                                    rocsparse_operation_none,
                                                                    0,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    buffer_size);
    }

    rocsparse_status null_csrsv_double_complex(rocsparse_handle handle, size_t* buffer_size)
    {
        return rocsparse_csrsv_buffer_size<rocsparse_double_complex>(handle,
                                                                     rocsparse_operation_none,
                                                                     0,
                                                                     0,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     buffer_size);
    }

    struct LogStreams
    {
        rocsparse_handle     handle;
        rocsparse_layer_mode saved_mode;
        std::ostream*        saved_trace;
        std::ostream*        saved_bench;
        std::ostream*        saved_debug;
        std::ostringstream   sink;

        explicit LogStreams(rocsparse_handle h)
            : handle(h)
            , saved_mode(h->layer_mode)
            , saved_trace(h->log_trace_os)
            , saved_bench(h->log_bench_os)
            , saved_debug(h->log_debug_os)
        {
            handle->log_trace_os = &sink;
            handle->log_bench_os = &sink;
            handle->log_debug_os = &sink;
            handle->layer_mode   = static_cast<rocsparse_layer_mode>(
                rocsparse_layer_mode_log_trace | rocsparse_layer_mode_log_bench
                | rocsparse_layer_mode_log_debug);
        }

        ~LogStreams()
        {
            handle->layer_mode   = saved_mode;
            handle->log_trace_os = saved_trace;
            handle->log_bench_os = saved_bench;
            handle->log_debug_os = saved_debug;
        }

        LogStreams(const LogStreams&) = delete;
        LogStreams& operator=(const LogStreams&) = delete;
    };

    rocsparse_int           g_index = 0;
    float                   g_value = 1.0f;
    constexpr rocsparse_int kM      = 1;
    constexpr rocsparse_int kNnz    = 1;

    rocsparse_status call_compute(rocsparse_handle     handle,
                                  rocsparse_int        m,
                                  rocsparse_int        nnz,
                                  const rocsparse_int* ptr_begin,
                                  const rocsparse_int* ptr_end,
                                  const rocsparse_int* ind,
                                  const float*         val,
                                  rocsparse_index_base base,
                                  rocsparse_index_base lbase,
                                  rocsparse_index_base ubase,
                                  rocsparse_diag_type  ldiag,
                                  rocsparse_diag_type  udiag,
                                  rocsparse_direction  ldir,
                                  rocsparse_direction  udir,
                                  rocsparse_int        lnnz,
                                  rocsparse_int        unnz,
                                  const rocsparse_int* lptr_begin,
                                  const rocsparse_int* lptr_end,
                                  const rocsparse_int* lind,
                                  float*               lval,
                                  const rocsparse_int* uptr_begin,
                                  const rocsparse_int* uptr_end,
                                  const rocsparse_int* uind,
                                  float*               uval,
                                  float*               dval,
                                  rocsparse_itilu0_alg alg,
                                  rocsparse_int*       nsweeps)
    {
        return rocsparse::csritilu0x_compute_impl<float, rocsparse_int, rocsparse_int>(handle,
                                                                                       alg,
                                                                                       0,
                                                                                       nsweeps,
                                                                                       0,
                                                                                       0.0f,
                                                                                       m,
                                                                                       nnz,
                                                                                       ptr_begin,
                                                                                       ptr_end,
                                                                                       ind,
                                                                                       val,
                                                                                       base,
                                                                                       ldiag,
                                                                                       ldir,
                                                                                       lnnz,
                                                                                       lptr_begin,
                                                                                       lptr_end,
                                                                                       lind,
                                                                                       lval,
                                                                                       lbase,
                                                                                       udiag,
                                                                                       udir,
                                                                                       unnz,
                                                                                       uptr_begin,
                                                                                       uptr_end,
                                                                                       uind,
                                                                                       uval,
                                                                                       ubase,
                                                                                       dval,
                                                                                       0,
                                                                                       nullptr);
    }

    rocsparse_status call_preprocess(rocsparse_handle     handle,
                                     rocsparse_int        m,
                                     rocsparse_int        nnz,
                                     const rocsparse_int* ptr_begin,
                                     const rocsparse_int* ptr_end,
                                     const rocsparse_int* ind,
                                     rocsparse_direction  ldir,
                                     rocsparse_direction  udir,
                                     rocsparse_int        lnnz,
                                     rocsparse_int        unnz,
                                     const rocsparse_int* lptr_begin,
                                     const rocsparse_int* lptr_end,
                                     const rocsparse_int* lind,
                                     const rocsparse_int* uptr_begin,
                                     const rocsparse_int* uptr_end,
                                     const rocsparse_int* uind,
                                     rocsparse_index_base base)
    {
        return rocsparse::csritilu0x_preprocess_impl<rocsparse_int, rocsparse_int>(
            handle,
            rocsparse_itilu0_alg_sync_split,
            0,
            0,
            m,
            nnz,
            ptr_begin,
            ptr_end,
            ind,
            base,
            rocsparse_diag_type_non_unit,
            ldir,
            lnnz,
            lptr_begin,
            lptr_end,
            lind,
            rocsparse_index_base_zero,
            rocsparse_diag_type_non_unit,
            udir,
            unnz,
            uptr_begin,
            uptr_end,
            uind,
            rocsparse_index_base_zero,
            rocsparse_datatype_f32_r,
            0,
            nullptr);
    }
}

TEST_F(HandleTest, template_wrappers_forward_to_typed_api)
{
    // The inline specializations in rocsparse_templates.hpp are never called by
    // the library. Zero sizes return before any device work, which still executes
    // every line of the wrapper. csrsv writes the buffer size on that path.
    // bsrmv, gebsrmv, and csrmm reject the null handle before any launch.
    size_t buffer_size = 1;
    EXPECT_EQ(null_csrsv(nullptr, &buffer_size), rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);
    EXPECT_EQ(null_csrsv_double(nullptr, &buffer_size), rocsparse_status_success);
    EXPECT_EQ(null_csrsv_float_complex(nullptr, &buffer_size), rocsparse_status_success);
    EXPECT_EQ(null_csrsv_double_complex(nullptr, &buffer_size), rocsparse_status_success);
    EXPECT_EQ(null_bsrmv<float>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_bsrmv<double>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_bsrmv<rocsparse_float_complex>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_bsrmv<rocsparse_double_complex>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_gebsrmv<float>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_gebsrmv<double>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_gebsrmv<rocsparse_float_complex>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_gebsrmv<rocsparse_double_complex>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_csrmm<float>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_csrmm<double>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_csrmm<rocsparse_float_complex>(nullptr), rocsparse_status_invalid_handle);
    EXPECT_EQ(null_csrmm<rocsparse_double_complex>(nullptr), rocsparse_status_invalid_handle);
}

TEST_F(HandleTest, swdev_csr_descr_argument_checks)
{
    const int debug_was_on = rocsparse_state_debug_arguments();

    rocsparse_spmat_descr descr = nullptr;
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      0,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);

    rocsparse_int row_ptr = 0;
    rocsparse_int col_ind = 0;
    float         value   = 1.0f;
    descr                 = nullptr;
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      1,
                                                      1,
                                                      1,
                                                      &row_ptr,
                                                      &col_ind,
                                                      &value,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);

    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(nullptr,
                                                      0,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      -1,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      1,
                                                      1,
                                                      2,
                                                      &row_ptr,
                                                      &col_ind,
                                                      &value,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      1,
                                                      1,
                                                      1,
                                                      nullptr,
                                                      &col_ind,
                                                      &value,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);

    rocsparse_enable_debug_arguments();
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(nullptr,
                                                      0,
                                                      0,
                                                      0,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_csr_descr_SWDEV_453599(&descr,
                                                      1,
                                                      1,
                                                      1,
                                                      &row_ptr,
                                                      &col_ind,
                                                      &value,
                                                      static_cast<rocsparse_indextype>(100),
                                                      rocsparse_indextype_i32,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_datatype_f32_r),
              rocsparse_status_invalid_value);
    if(debug_was_on == 0)
    {
        rocsparse_disable_debug_arguments();
    }
}

TEST_F(HandleTest, logging_streams_and_layer_mode)
{
    std::ostream* cerr_stream = nullptr;
    std::ofstream file;
    rocsparse::open_log_stream(&cerr_stream, &file, "ROCSPARSE_COV_LOG_UNSET");
    EXPECT_EQ(cerr_stream, &std::cerr);

    const std::string path = temp_file_path("rocsparse-cov-gap-log.txt");
    set_env("ROCSPARSE_COV_LOG_FILE", path.c_str());
    rocsparse::open_log_stream(&cerr_stream, &file, "ROCSPARSE_COV_LOG_FILE");
    EXPECT_TRUE(file.is_open());
    file.close();

    set_env("ROCSPARSE_COV_LOG_FILE", "/no/such/dir/rocsparse-cov-gap-log.txt");
    std::ofstream missing;
    rocsparse::open_log_stream(&cerr_stream, &missing, "ROCSPARSE_COV_LOG_FILE");
    EXPECT_FALSE(missing.is_open());
    EXPECT_EQ(cerr_stream, &std::cerr);
    unset_env("ROCSPARSE_COV_LOG_FILE");

    EXPECT_NE(rocsparse::replaceX<float>("rocsparse_Xcov").find('s'), std::string::npos);
    EXPECT_NE(rocsparse::replaceX<double>("rocsparse_Xcov").find('d'), std::string::npos);

    rocsparse::log_trace(static_cast<rocsparse_handle>(nullptr), "cov_trace_null");

    {
        LogStreams logs(handle);
        rocsparse::log_trace(handle, "cov_trace", 1, 2.0f);
        rocsparse::log_bench(handle, "cov_bench", std::string("s"), 1);
        rocsparse::log_debug(handle, "cov_debug");

        const float host_value = 1.5f;
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(&host_value), host_value);
        EXPECT_TRUE(
            std::isnan(rocsparse::log_trace_scalar_value(static_cast<const float*>(nullptr))));
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(handle, &host_value), host_value);
        EXPECT_TRUE(std::isnan(
            rocsparse::log_trace_scalar_value(handle, static_cast<const float*>(nullptr))));
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(&host_value), host_value);
        EXPECT_TRUE(
            std::isnan(rocsparse::log_bench_scalar_value(static_cast<const float*>(nullptr))));
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(handle, &host_value), host_value);

        float* device_value = nullptr;
        ASSERT_EQ(hipMalloc(&device_value, sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemcpy(device_value, &host_value, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
        rocsparse_pointer_mode saved_mode = rocsparse_pointer_mode_host;
        ASSERT_EQ(rocsparse_get_pointer_mode(handle, &saved_mode), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                  rocsparse_status_success);
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(handle, device_value), host_value);
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(handle, device_value), host_value);
        EXPECT_EQ(rocsparse_set_pointer_mode(handle, saved_mode), rocsparse_status_success);
        EXPECT_EQ(hipFree(device_value), hipSuccess);
        EXPECT_FALSE(logs.sink.str().empty());
    }
}

TEST_F(HandleTest, itilu0x_compute_argument_checks)
{
    rocsparse_int sweeps = 1;
    EXPECT_EQ(call_compute(nullptr,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(call_compute(handle,
                           -1,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call_compute(handle,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_success);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_success);

    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           nullptr,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           nullptr,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           nullptr,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           nullptr,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);

    const auto bad_base = static_cast<rocsparse_index_base>(100);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           bad_base,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_value);

    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           nullptr,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           1,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           1,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_unit,
                           rocsparse_diag_type_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           nullptr,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_column,
                           rocsparse_direction_column,
                           0,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_value,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_not_implemented);
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_row,
                           0,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_value,
                           rocsparse_itilu0_alg_sync_split,
                           &sweeps),
              rocsparse_status_not_implemented);

    // async_inplace is rejected inside the dispatcher, before any driver runs.
    EXPECT_EQ(call_compute(handle,
                           kM,
                           kNnz,
                           &g_index,
                           &g_index,
                           &g_index,
                           &g_value,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_index_base_zero,
                           rocsparse_diag_type_non_unit,
                           rocsparse_diag_type_non_unit,
                           rocsparse_direction_row,
                           rocsparse_direction_column,
                           0,
                           0,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_index,
                           &g_index,
                           nullptr,
                           nullptr,
                           &g_value,
                           rocsparse_itilu0_alg_async_inplace,
                           &sweeps),
              rocsparse_status_internal_error);
}

TEST_F(HandleTest, itilu0x_preprocess_argument_checks)
{
    EXPECT_EQ(call_preprocess(nullptr,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(call_preprocess(handle,
                              -1,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_invalid_size);
    EXPECT_EQ(call_preprocess(handle,
                              kM,
                              0,
                              nullptr,
                              &g_index,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_preprocess(handle,
                              kM,
                              kNnz,
                              &g_index,
                              &g_index,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              &g_index,
                              &g_index,
                              nullptr,
                              &g_index,
                              &g_index,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(call_preprocess(handle,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_column,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_not_implemented);
    EXPECT_EQ(call_preprocess(handle,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_row,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_not_implemented);
    EXPECT_EQ(call_preprocess(handle,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              static_cast<rocsparse_index_base>(100)),
              rocsparse_status_invalid_value);

    // m > 0 and nnz == 0 reaches the template and returns before the driver.
    EXPECT_EQ(call_preprocess(handle,
                              kM,
                              0,
                              &g_index,
                              &g_index,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              &g_index,
                              &g_index,
                              nullptr,
                              &g_index,
                              &g_index,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_zero_pivot);
    EXPECT_EQ(call_preprocess(handle,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_direction_row,
                              rocsparse_direction_column,
                              0,
                              0,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              rocsparse_index_base_zero),
              rocsparse_status_success);
}

TEST_F(HandleTest, itilu0x_buffer_size_argument_checks)
{
    size_t bytes = 1;
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  nullptr,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  0,
                  0,
                  nullptr,
                  nullptr,
                  nullptr,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_invalid_handle);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  -1,
                  0,
                  nullptr,
                  nullptr,
                  nullptr,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_invalid_size);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  kM,
                  kNnz,
                  nullptr,
                  &g_index,
                  &g_index,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  kM,
                  kNnz,
                  &g_index,
                  &g_index,
                  nullptr,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  0,
                  0,
                  nullptr,
                  nullptr,
                  nullptr,
                  static_cast<rocsparse_index_base>(100),
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_invalid_value);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  0,
                  0,
                  nullptr,
                  nullptr,
                  nullptr,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, 0u);
    EXPECT_EQ((rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                  handle,
                  rocsparse_itilu0_alg_sync_split,
                  0,
                  0,
                  kM,
                  0,
                  nullptr,
                  nullptr,
                  nullptr,
                  rocsparse_index_base_zero,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_row,
                  rocsparse_diag_type_non_unit,
                  rocsparse_direction_column,
                  rocsparse_datatype_f32_r,
                  &bytes)),
              rocsparse_status_zero_pivot);
}
