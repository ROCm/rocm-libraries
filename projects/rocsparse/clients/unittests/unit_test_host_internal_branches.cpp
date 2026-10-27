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

// Host paths of the library internals that the public clients never execute:
//   - the inline template wrappers of rocsparse_templates.hpp, the SWDEV CSR
//     descriptor creator and the ITILU0 x-impl argument checks,
//   - handle construction with logging, the logging helpers, the
//     argument-descriptor getters, the hip status translation and the debug
//     switches,
//   - the CSR triangular split and copy_csrmv_info.
// Coverage counts a line only when every branch of that line has run, so each
// check is taken on both sides. Only the CSR triangular split launches a
// (tiny) kernel; the ITILU0 x-impls return from argument checks, from the
// m == 0 / nnz == 0 shortcuts, or from the async-inplace dispatch (that case
// returns internal_error before the driver).

#include "unit_test_host_utils.hpp"

#include "rocsparse-debugging.h"
#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_templates.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <string>
#include <vector>

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

extern "C" rocsparse_status rocsparse_argdescr_create(void** argdescr);
extern "C" rocsparse_status rocsparse_argdescr_free(void* argdescr);
extern "C" rocsparse_status rocsparse_argdescr_get_msg(const void* argdescr, const char** res);
extern "C" rocsparse_status rocsparse_argdescr_get_status(const void*       argdescr,
                                                          rocsparse_status* res);
extern "C" rocsparse_status rocsparse_argdescr_get_tid(const void* argdescr, int64_t* tid);
extern "C" rocsparse_status rocsparse_argdescr_get_pid(const void* argdescr, int64_t* pid);
extern "C" rocsparse_status rocsparse_argdescr_get_index(const void* argdescr, int* index);
extern "C" rocsparse_status rocsparse_argdescr_get_name(const void* argdescr, const char** name);
extern "C" rocsparse_status rocsparse_argdescr_get_function_line(const void* argdescr, int* line);
extern "C" rocsparse_status rocsparse_argdescr_get_function_name(const void*  argdescr,
                                                                 const char** name);

namespace
{
    // The inline specializations in rocsparse_templates.hpp are never called by
    // the library (AISPARSE-921). Zero sizes return before any device work, which still
    // executes every line of the wrapper. bsrmv, gebsrmv and csrmm reject the
    // null handle before any launch. The csrsv header documents invalid_handle,
    // but the m == 0 quick return runs before the handle check and succeeds
    // (AISPARSE-929), so both are accepted; a success must still write the (empty)
    // buffer size.
    template <typename T>
    void expect_template_wrappers_reject_null_handle()
    {
        size_t     buffer_size = 1;
        const auto status      = rocsparse_csrsv_buffer_size<T>(nullptr,
                                                           rocsparse_operation_none,
                                                           0,
                                                           0,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           &buffer_size);
        EXPECT_TRUE(status == rocsparse_status_invalid_handle || status == rocsparse_status_success)
            << status;
        if(status == rocsparse_status_success)
        {
            EXPECT_EQ(buffer_size, size_t(0));
        }

        EXPECT_EQ(rocsparse_bsrmv<T>(nullptr,
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
                                     nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_gebsrmv<T>(nullptr,
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
                                       nullptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_csrmm<T>(nullptr,
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
                                     0),
                  rocsparse_status_invalid_handle);
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

    rocsparse_int g_index = 0;
    float         g_value = 1.0f;

    // Arguments of the three ITILU0 x-impls, with the m == 0 defaults. Each
    // check changes only the fields it needs. The factors' index base and
    // diagonal types are shared by compute, preprocess and buffer_size; the
    // fields a given impl does not take are ignored by its wrapper below.
    struct Itilu0Args
    {
        rocsparse_handle     handle     = nullptr;
        rocsparse_itilu0_alg alg        = rocsparse_itilu0_alg_sync_split;
        rocsparse_int        m          = 0;
        rocsparse_int        nnz        = 0;
        const rocsparse_int* ptr_begin  = nullptr;
        const rocsparse_int* ptr_end    = nullptr;
        const rocsparse_int* ind        = nullptr;
        const float*         val        = nullptr;
        rocsparse_index_base base       = rocsparse_index_base_zero;
        rocsparse_diag_type  ldiag      = rocsparse_diag_type_non_unit;
        rocsparse_diag_type  udiag      = rocsparse_diag_type_non_unit;
        rocsparse_direction  ldir       = rocsparse_direction_row;
        rocsparse_direction  udir       = rocsparse_direction_column;
        rocsparse_int        lnnz       = 0;
        rocsparse_int        unnz       = 0;
        const rocsparse_int* lptr_begin = nullptr;
        const rocsparse_int* lptr_end   = nullptr;
        const rocsparse_int* lind       = nullptr;
        float*               lval       = nullptr;
        const rocsparse_int* uptr_begin = nullptr;
        const rocsparse_int* uptr_end   = nullptr;
        const rocsparse_int* uind       = nullptr;
        float*               uval       = nullptr;
        float*               dval       = nullptr;

        // One non-zero in A, with valid pointers for A only.
        Itilu0Args& with_matrix()
        {
            m = nnz   = 1;
            ptr_begin = ptr_end = ind = &g_index;
            val                       = &g_value;
            return *this;
        }

        // Also valid row-pointer arrays for both factors (their index arrays
        // and values stay null).
        Itilu0Args& with_factors()
        {
            with_matrix();
            lptr_begin = lptr_end = uptr_begin = uptr_end = &g_index;
            return *this;
        }

        rocsparse_status compute()
        {
            rocsparse_int sweeps = 1;
            return rocsparse::csritilu0x_compute_impl<float, rocsparse_int, rocsparse_int>(
                handle,
                alg,
                0,
                &sweeps,
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
                rocsparse_index_base_zero,
                udiag,
                udir,
                unnz,
                uptr_begin,
                uptr_end,
                uind,
                uval,
                rocsparse_index_base_zero,
                dval,
                0,
                nullptr);
        }

        rocsparse_status preprocess()
        {
            return rocsparse::csritilu0x_preprocess_impl<rocsparse_int, rocsparse_int>(
                handle,
                alg,
                0,
                0,
                m,
                nnz,
                ptr_begin,
                ptr_end,
                ind,
                base,
                ldiag,
                ldir,
                lnnz,
                lptr_begin,
                lptr_end,
                lind,
                rocsparse_index_base_zero,
                udiag,
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

        rocsparse_status buffer_size(size_t* bytes)
        {
            return rocsparse::csritilu0x_buffer_size_impl<rocsparse_int, rocsparse_int>(
                handle,
                alg,
                0,
                0,
                m,
                nnz,
                ptr_begin,
                ptr_end,
                ind,
                base,
                ldiag,
                ldir,
                udiag,
                udir,
                rocsparse_datatype_f32_r,
                bytes);
        }
    };

    const auto bad_base = static_cast<rocsparse_index_base>(100);
}

namespace
{
    struct scoped_stream
    {
        hipStream_t stream = nullptr;

        scoped_stream() = default;
        ~scoped_stream()
        {
            if(stream != nullptr)
            {
                EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
            }
        }

        scoped_stream(const scoped_stream&) = delete;
        scoped_stream& operator=(const scoped_stream&) = delete;
    };

    void set_arch_name(hipDeviceProp_t& prop, const char* name)
    {
        std::strncpy(prop.gcnArchName, name, sizeof(prop.gcnArchName) - 1);
        prop.gcnArchName[sizeof(prop.gcnArchName) - 1] = '\0';
    }

    // Creates and destroys a handle through the public (stream == nullptr) or
    // the internal (explicit stream) constructor.
    rocsparse_status open_and_close(hipStream_t stream)
    {
        rocsparse_handle logged = nullptr;
        rocsparse_status status = (stream == nullptr)
                                      ? rocsparse_create_handle(&logged)
                                      : rocsparse_handle_create(&logged, stream, nullptr);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        return (stream == nullptr) ? rocsparse_destroy_handle(logged)
                                   : rocsparse_handle_destroy(logged, nullptr);
    }

    // Calls every argument-descriptor getter with either null or real outputs.
    // A live descriptor holds the record of rocsparse_create_handle(nullptr).
    void expect_argdescr_getters(const void* descr, bool null_outputs)
    {
        const char*      msg           = nullptr;
        rocsparse_status status        = rocsparse_status_success;
        int64_t          tid           = -1;
        int64_t          pid           = -1;
        int              index         = -1;
        const char*      name          = nullptr;
        int              function_line = -1;
        const char*      function_name = nullptr;

        const rocsparse_status want = (descr == nullptr || null_outputs)
                                          ? rocsparse_status_invalid_pointer
                                          : rocsparse_status_success;
        auto                   out  = [&](auto& value) { return null_outputs ? nullptr : &value; };

        EXPECT_EQ(rocsparse_argdescr_get_msg(descr, out(msg)), want);
        EXPECT_EQ(rocsparse_argdescr_get_status(descr, out(status)), want);
        EXPECT_EQ(rocsparse_argdescr_get_tid(descr, out(tid)), want);
        EXPECT_EQ(rocsparse_argdescr_get_pid(descr, out(pid)), want);
        EXPECT_EQ(rocsparse_argdescr_get_index(descr, out(index)), want);
        EXPECT_EQ(rocsparse_argdescr_get_name(descr, out(name)), want);
        EXPECT_EQ(rocsparse_argdescr_get_function_line(descr, out(function_line)), want);
        // rocsparse_argdescr_get_function_name does not check its output pointer.
        if(descr == nullptr || !null_outputs)
        {
            EXPECT_EQ(rocsparse_argdescr_get_function_name(descr, out(function_name)), want);
        }
        if(want == rocsparse_status_success)
        {
            ASSERT_NE(msg, nullptr);
            ASSERT_NE(name, nullptr);
            ASSERT_NE(function_name, nullptr);
            EXPECT_NE(std::string(msg).find("null"), std::string::npos) << msg;
            EXPECT_EQ(status, rocsparse_status_invalid_pointer);
            EXPECT_EQ(index, 0);
            EXPECT_STREQ(name, "handle");
            EXPECT_NE(std::string(function_name).find("rocsparse_create_handle"), std::string::npos)
                << function_name;
            EXPECT_GT(function_line, 0);
        }
    }

    rocsparse_status sldu_buffer_size(rocsparse_handle    handle,
                                      rocsparse_direction dir,
                                      rocsparse_direction ldir,
                                      rocsparse_direction udir,
                                      size_t*             bytes)
    {
        return rocsparse::csxsldu_buffer_size_template<float, rocsparse_int, rocsparse_int>(
            handle,
            dir,
            4,
            4,
            4,
            nullptr,
            nullptr,
            nullptr,
            rocsparse_index_base_zero,
            rocsparse_diag_type_unit,
            ldir,
            rocsparse_diag_type_unit,
            udir,
            bytes);
    }

    rocsparse_status sldu_preprocess(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_diag_type         ldiag,
                                     rocsparse_direction         ldir,
                                     rocsparse_diag_type         udiag,
                                     rocsparse_direction         udir,
                                     rocsparse_int*              scratch,
                                     rocsparse_int*              lnnz_out,
                                     rocsparse_int*              unnz_out,
                                     std::vector<rocsparse_int>* lptr_out = nullptr,
                                     std::vector<rocsparse_int>* uptr_out = nullptr)
    {
        const std::vector<rocsparse_int> sentinel{-7, -7, -7};
        device_vector<rocsparse_int>     ptr{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int>     ind{std::vector<rocsparse_int>{0, 1}};
        device_vector<float>             val{std::vector<float>{1.0f, 1.0f}};
        device_vector<rocsparse_int>     lptr{sentinel};
        device_vector<rocsparse_int>     uptr{sentinel};
        if(!ptr.ptr || !ind.ptr || !val.ptr || !lptr.ptr || !uptr.ptr)
        {
            return rocsparse_status_memory_error;
        }
        rocsparse_int          lnnz = -1;
        rocsparse_int          unnz = -1;
        const rocsparse_status status
            = rocsparse::csxsldu_preprocess_template<float, rocsparse_int, rocsparse_int>(
                handle,
                dir,
                2,
                2,
                2,
                ptr.ptr,
                ind.ptr,
                val.ptr,
                rocsparse_index_base_zero,
                ldiag,
                ldir,
                &lnnz,
                lptr.ptr,
                rocsparse_index_base_zero,
                udiag,
                udir,
                &unnz,
                uptr.ptr,
                rocsparse_index_base_zero,
                scratch);
        if(lnnz_out)
            *lnnz_out = lnnz;
        if(unnz_out)
            *unnz_out = unnz;
        if(lptr_out)
            *lptr_out = to_host(lptr);
        if(uptr_out)
            *uptr_out = to_host(uptr);
        return status;
    }
    class HostInternalBranches : public HandleTest
    {
    };
} // namespace

TEST_F(HostInternalBranches, template_wrappers_reject_null_handle)
{
    expect_template_wrappers_reject_null_handle<float>();
    expect_template_wrappers_reject_null_handle<double>();
    expect_template_wrappers_reject_null_handle<rocsparse_float_complex>();
    expect_template_wrappers_reject_null_handle<rocsparse_double_complex>();
}

TEST_F(HostInternalBranches, swdev_csr_descr_argument_checks)
{
    const debug_state_guard restore;

    int32_t row_ptr = 0;
    int32_t col_ind = 0;
    float   value   = 1.0f;

    auto create = [&](rocsparse_spmat_descr* d,
                      int64_t                rows,
                      int64_t                cols,
                      int64_t                nnz,
                      void*                  ptr,
                      void*                  ind,
                      void*                  val,
                      rocsparse_indextype    row_ptr_type = rocsparse_indextype_i32) {
        return rocsparse_create_csr_descr_SWDEV_453599(d,
                                                       rows,
                                                       cols,
                                                       nnz,
                                                       ptr,
                                                       ind,
                                                       val,
                                                       row_ptr_type,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       rocsparse_datatype_f32_r);
    };

    // Empty and 1x1 matrices are created, read back and destroyed.
    auto expect_csr = [](rocsparse_spmat_descr d, int64_t size, void* ptr, void* ind, void* val) {
        int64_t              rows = -1, cols = -1, nnz = -1;
        void *               p = &rows, *i = &rows, *v = &rows;
        rocsparse_indextype  pt = rocsparse_indextype_i64, it = rocsparse_indextype_i64;
        rocsparse_index_base base = rocsparse_index_base_one;
        rocsparse_datatype   dt   = rocsparse_datatype_f64_r;
        ASSERT_EQ(rocsparse_csr_get(d, &rows, &cols, &nnz, &p, &i, &v, &pt, &it, &base, &dt),
                  rocsparse_status_success);
        EXPECT_EQ(rows, size);
        EXPECT_EQ(cols, size);
        EXPECT_EQ(nnz, size);
        EXPECT_EQ(p, ptr);
        EXPECT_EQ(i, ind);
        EXPECT_EQ(v, val);
        EXPECT_EQ(pt, rocsparse_indextype_i32);
        EXPECT_EQ(it, rocsparse_indextype_i32);
        EXPECT_EQ(base, rocsparse_index_base_zero);
        EXPECT_EQ(dt, rocsparse_datatype_f32_r);
    };
    rocsparse_spmat_descr descr = nullptr;
    EXPECT_EQ(create(&descr, 0, 0, 0, nullptr, nullptr, nullptr), rocsparse_status_success);
    expect_csr(descr, 0, nullptr, nullptr, nullptr);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
    descr = nullptr;
    EXPECT_EQ(create(&descr, 1, 1, 1, &row_ptr, &col_ind, &value), rocsparse_status_success);
    expect_csr(descr, 1, &row_ptr, &col_ind, &value);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);

    descr = nullptr;
    EXPECT_EQ(create(nullptr, 0, 0, 0, nullptr, nullptr, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(create(&descr, -1, 0, 0, nullptr, nullptr, nullptr), rocsparse_status_invalid_size);
    EXPECT_EQ(create(&descr, 1, 1, 2, &row_ptr, &col_ind, &value), rocsparse_status_invalid_size);
    EXPECT_EQ(create(&descr, 1, 1, 1, nullptr, &col_ind, &value), rocsparse_status_invalid_pointer);

    rocsparse_enable_debug_arguments();
    EXPECT_EQ(create(nullptr, 0, 0, 0, nullptr, nullptr, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        create(&descr, 1, 1, 1, &row_ptr, &col_ind, &value, static_cast<rocsparse_indextype>(100)),
        rocsparse_status_invalid_value);
}

TEST_F(HostInternalBranches, logging_streams_and_layer_mode)
{
    constexpr const char* path_var    = "ROCSPARSE_UT_LOG_PATH";
    std::ostream*         cerr_stream = nullptr;
    {
        const scoped_env unset(path_var, "");
        unset_env(path_var);
        std::ofstream file;
        rocsparse::open_log_stream(&cerr_stream, &file, path_var);
        EXPECT_EQ(cerr_stream, &std::cerr);
        EXPECT_FALSE(file.is_open());
    }
    {
        const scoped_temp_file log("rocsparse-ut-log.txt");
        const scoped_env       path(path_var, log.path.c_str());
        std::ofstream          file;
        rocsparse::open_log_stream(&cerr_stream, &file, path_var);
        EXPECT_TRUE(file.is_open());
        EXPECT_EQ(cerr_stream, &file);
    }
    {
        const scoped_env path(path_var, "/no/such/dir/rocsparse-ut-log.txt");
        std::ofstream    missing;
        rocsparse::open_log_stream(&cerr_stream, &missing, path_var);
        EXPECT_FALSE(missing.is_open());
        EXPECT_EQ(cerr_stream, &std::cerr);
    }

    EXPECT_EQ(rocsparse::replaceX<float>("rocsparse_Xtest"), "rocsparse_stest");
    EXPECT_EQ(rocsparse::replaceX<double>("rocsparse_Xtest"), "rocsparse_dtest");

    rocsparse::log_trace(static_cast<rocsparse_handle>(nullptr), "ut_trace_null");

    {
        LogStreams logs(handle);
        rocsparse::log_trace(handle, "ut_trace", 1, 2.0f);
        rocsparse::log_bench(handle, "ut_bench", std::string("s"), 1);
        rocsparse::log_debug(handle, "ut_debug");

        // Both scalar loggers read a host value, return NaN for a null pointer
        // and, in device pointer mode, copy the value back from the device.
        const float  host_value = 1.5f;
        const float* null_value = nullptr;
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(&host_value), host_value);
        EXPECT_TRUE(std::isnan(rocsparse::log_trace_scalar_value(null_value)));
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(handle, &host_value), host_value);
        EXPECT_TRUE(std::isnan(rocsparse::log_trace_scalar_value(handle, null_value)));
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(&host_value), host_value);
        EXPECT_TRUE(std::isnan(rocsparse::log_bench_scalar_value(null_value)));
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(handle, &host_value), host_value);

        device_vector<float> device_value{std::vector<float>{host_value}};
        ASSERT_TRUE(device_value.ptr);
        rocsparse_pointer_mode saved_mode = rocsparse_pointer_mode_host;
        ASSERT_EQ(rocsparse_get_pointer_mode(handle, &saved_mode), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                  rocsparse_status_success);
        EXPECT_FLOAT_EQ(rocsparse::log_trace_scalar_value(handle, device_value.ptr), host_value);
        EXPECT_FLOAT_EQ(rocsparse::log_bench_scalar_value(handle, device_value.ptr), host_value);
        EXPECT_EQ(rocsparse_set_pointer_mode(handle, saved_mode), rocsparse_status_success);
        const std::string text = logs.sink.str();
        for(const char* head : {"ut_trace", "ut_bench", "ut_debug"})
        {
            EXPECT_NE(text.find(head), std::string::npos) << head;
        }
    }
}

TEST_F(HostInternalBranches, itilu0x_compute_argument_checks)
{
    constexpr auto invalid_pointer = rocsparse_status_invalid_pointer;

    auto check = [&](rocsparse_status expected, auto&& edit) {
        Itilu0Args a;
        a.handle = handle;
        edit(a);
        EXPECT_EQ(a.compute(), expected);
    };
    auto check_fails = [&](auto&& edit) {
        Itilu0Args a;
        a.handle = handle;
        edit(a);
        EXPECT_NE(a.compute(), rocsparse_status_success);
    };

    Itilu0Args null_handle;
    EXPECT_EQ(null_handle.compute(), rocsparse_status_invalid_handle);
    check(rocsparse_status_invalid_size, [](Itilu0Args& a) { a.m = -1; });
    check(rocsparse_status_success, [](Itilu0Args&) {});
    check(rocsparse_status_success, [](Itilu0Args& a) { a.m = 1; });

    // Each array of A is checked when nnz > 0.
    for(auto member : {&Itilu0Args::ptr_begin, &Itilu0Args::ptr_end, &Itilu0Args::ind})
    {
        check(invalid_pointer, [&](Itilu0Args& a) { a.with_matrix().*member = nullptr; });
    }
    check(invalid_pointer, [](Itilu0Args& a) { a.with_matrix().val = nullptr; });

    // With a valid A: index base, factor pointers and diagonal types.
    check(rocsparse_status_invalid_value, [](Itilu0Args& a) { a.with_factors().base = bad_base; });
    check(invalid_pointer, [](Itilu0Args& a) { a.with_factors().lptr_begin = nullptr; });
    check(invalid_pointer, [](Itilu0Args& a) { a.with_factors().unnz = 1; });
    check(invalid_pointer, [](Itilu0Args& a) {
        a.with_factors();
        a.lnnz = 1;
        a.uind = &g_index;
        a.uval = &g_value;
    });
    check(invalid_pointer,
          [](Itilu0Args& a) { a.with_factors().ldiag = a.udiag = rocsparse_diag_type_unit; });

    // L by rows with U by columns is the only supported layout, and
    // async_inplace is rejected inside the dispatcher, before any driver runs.
    // The headers document neither status, so only a failure is checked.
    check_fails([](Itilu0Args& a) {
        a.with_factors().dval = &g_value;
        a.ldir = a.udir = rocsparse_direction_column;
    });
    check_fails([](Itilu0Args& a) {
        a.with_factors().dval = &g_value;
        a.ldir = a.udir = rocsparse_direction_row;
    });
    check_fails([](Itilu0Args& a) {
        a.with_factors().dval = &g_value;
        a.alg                 = rocsparse_itilu0_alg_async_inplace;
    });
}

TEST_F(HostInternalBranches, itilu0x_preprocess_argument_checks)
{
    auto check = [&](rocsparse_status expected, auto&& edit) {
        Itilu0Args a;
        a.handle = handle;
        edit(a);
        EXPECT_EQ(a.preprocess(), expected);
    };
    auto check_fails = [&](auto&& edit) {
        Itilu0Args a;
        a.handle = handle;
        edit(a);
        EXPECT_NE(a.preprocess(), rocsparse_status_success);
    };

    Itilu0Args null_handle;
    EXPECT_EQ(null_handle.preprocess(), rocsparse_status_invalid_handle);
    check(rocsparse_status_invalid_size, [](Itilu0Args& a) { a.m = -1; });
    check(rocsparse_status_success, [](Itilu0Args&) {});
    check(rocsparse_status_invalid_value, [](Itilu0Args& a) { a.base = bad_base; });
    // Unsupported factor layouts; the header does not document the status.
    check_fails([](Itilu0Args& a) { a.ldir = a.udir = rocsparse_direction_column; });
    check_fails([](Itilu0Args& a) { a.ldir = a.udir = rocsparse_direction_row; });
    check(rocsparse_status_invalid_pointer, [](Itilu0Args& a) {
        a.m       = 1;
        a.ptr_end = &g_index;
    });
    check(rocsparse_status_invalid_pointer, [](Itilu0Args& a) { a.with_factors().ind = nullptr; });
    // m > 0 and nnz == 0 reaches the template and returns before the driver.
    check(rocsparse_status_zero_pivot, [](Itilu0Args& a) { a.with_factors().nnz = 0; });
}

TEST_F(HostInternalBranches, itilu0x_buffer_size_argument_checks)
{
    size_t bytes = 1;
    auto   check = [&](rocsparse_status expected, auto&& edit) {
        Itilu0Args a;
        a.handle = handle;
        edit(a);
        EXPECT_EQ(a.buffer_size(&bytes), expected);
    };

    Itilu0Args null_handle;
    EXPECT_EQ(null_handle.buffer_size(&bytes), rocsparse_status_invalid_handle);
    check(rocsparse_status_invalid_size, [](Itilu0Args& a) { a.m = -1; });
    check(rocsparse_status_invalid_value, [](Itilu0Args& a) { a.base = bad_base; });
    check(rocsparse_status_zero_pivot, [](Itilu0Args& a) { a.m = 1; });
    check(rocsparse_status_invalid_pointer,
          [](Itilu0Args& a) { a.with_matrix().ptr_begin = nullptr; });
    check(rocsparse_status_invalid_pointer, [](Itilu0Args& a) { a.with_matrix().ind = nullptr; });

    check(rocsparse_status_success, [](Itilu0Args&) {});
    EXPECT_EQ(bytes, 0u);
}

// Both handle constructors open trace, bench and debug logs only when
// ROCSPARSE_LAYER has those bits. The public clients leave the variable unset.
TEST_F(HostInternalBranches, handle_layer_logs_and_arch)
{
    const char* const log_vars[]
        = {"ROCSPARSE_LOG_TRACE_PATH", "ROCSPARSE_LOG_BENCH_PATH", "ROCSPARSE_LOG_DEBUG_PATH"};
    const scoped_env layer("ROCSPARSE_LAYER", "7");
    const scoped_env restore_trace(log_vars[0], ""), restore_bench(log_vars[1], ""),
        restore_debug(log_vars[2], "");
    for(const char* var : log_vars)
    {
        unset_env(var);
    }

    scoped_stream owned_stream;
    ASSERT_EQ(hipStreamCreate(&owned_stream.stream), hipSuccess);
    const hipStream_t stream = owned_stream.stream;
    // Paths unset: open_log_stream keeps std::cerr.
    EXPECT_EQ(open_and_close(nullptr), rocsparse_status_success);
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);

    const scoped_temp_file files[] = {scoped_temp_file("rocsparse-ut-log-trace.txt"),
                                      scoped_temp_file("rocsparse-ut-log-bench.txt"),
                                      scoped_temp_file("rocsparse-ut-log-debug.txt")};
    for(int i = 0; i < 3; ++i)
    {
        set_env(log_vars[i], files[i].path.c_str());
    }

    {
        Owned<rocsparse_handle, rocsparse_destroy_handle> logged;
        ASSERT_EQ(rocsparse_create_handle(&logged.get), rocsparse_status_success);
        rocsparse::log_trace(nullptr, "ut_log");
        rocsparse::log_trace(handle, "ut_log");
        rocsparse::log_trace(logged.get, "ut_log");
        rocsparse::log_trace(logged.get,
                             "ut_log",
                             rocsparse_float_complex{1.0f, 2.0f},
                             rocsparse_double_complex{1.0, 2.0},
                             static_cast<_Float16>(1.0f),
                             rocsparse_bfloat16{1.0f});
        rocsparse::log_bench(nullptr, "ut_log", std::string("s"));
        rocsparse::log_bench(
            logged.get, "ut_log", std::string("s"), rocsparse_float_complex{1.0f, 2.0f});
        rocsparse::log_debug(nullptr, "ut_log");
        rocsparse::log_debug(logged.get, "ut_log");
    }

    for(const auto& file : files)
    {
        std::ifstream     in(file.path);
        const std::string text((std::istreambuf_iterator<char>(in)), {});
        EXPECT_NE(text.find("ut_log"), std::string::npos) << file.path;
    }
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);

    // A missing directory leaves the ofstream closed, so the cerr fallback runs.
    const std::string missing[] = {"/no/such/dir/rocsparse-ut-log-trace.txt",
                                   "/no/such/dir/rocsparse-ut-log-bench.txt",
                                   "/no/such/dir/rocsparse-ut-log-debug.txt"};
    for(int i = 0; i < 3; ++i)
    {
        set_env(log_vars[i], missing[i].c_str());
    }
    EXPECT_EQ(open_and_close(nullptr), rocsparse_status_success);
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);
    for(const auto& path : missing)
    {
        EXPECT_FALSE(std::ifstream(path).good()) << path;
    }

    auto*           raw   = reinterpret_cast<_rocsparse_handle*>(handle);
    hipDeviceProp_t saved = raw->properties;
    set_arch_name(raw->properties, "gfx90a:xnack-");
    EXPECT_EQ(rocsparse::handle_get_xnack_mode(handle), "xnack-");
    EXPECT_EQ(rocsparse::handle_get_arch_name(handle), "gfx90a");
    set_arch_name(raw->properties, "xnack");
    EXPECT_EQ(rocsparse::handle_get_xnack_mode(handle), "");
    set_arch_name(raw->properties, "gfx1201");
    EXPECT_EQ(rocsparse::handle_get_xnack_mode(handle), "");
    raw->properties = saved;
}

// Every hipError_t arm, and every catch in the C++ exception translator. The
// table is the library's internal mapping, not a documented contract.
TEST_F(HostInternalBranches, hip_status_and_exceptions)
{
    const std::pair<hipError_t, rocsparse_status> table[]
        = {{hipSuccess, rocsparse_status_success},
           {hipErrorMemoryAllocation, rocsparse_status_memory_error},
           {hipErrorLaunchOutOfResources, rocsparse_status_memory_error},
           {hipErrorInvalidDevicePointer, rocsparse_status_invalid_pointer},
           {hipErrorInvalidDevice, rocsparse_status_invalid_handle},
           {hipErrorInvalidResourceHandle, rocsparse_status_invalid_handle},
           {hipErrorInvalidValue, rocsparse_status_internal_error},
           {hipErrorNoDevice, rocsparse_status_internal_error},
           {hipErrorUnknown, rocsparse_status_internal_error},
           {hipErrorNotReady, rocsparse_status_internal_error}};
    for(const auto& [error, expected] : table)
    {
        EXPECT_EQ(rocsparse::get_rocsparse_status_for_hip_status(error), expected) << error;
    }

    auto status_of = [](auto thrown) {
        try
        {
            throw thrown;
        }
        catch(...)
        {
            return rocsparse::exception_to_rocsparse_status(std::current_exception());
        }
    };
    EXPECT_EQ(rocsparse::exception_to_rocsparse_status(nullptr), rocsparse_status_success);
    EXPECT_EQ(status_of(rocsparse_status_invalid_pointer), rocsparse_status_invalid_pointer);
    EXPECT_EQ(status_of(std::bad_alloc()), rocsparse_status_memory_error);
    EXPECT_EQ(status_of(1), rocsparse_status_thrown_exception);
}

// The argument-descriptor getters are exported but the clients only read the
// name and the index. Verbose mode prints the descriptor that a bad argument
// just recorded.
TEST_F(HostInternalBranches, argdescr_getters_and_verbose)
{
    const debug_state_guard restore;
    rocsparse_enable_debug_arguments();
    rocsparse_enable_debug_arguments_verbose();
    // Once before the descriptor exists (the stack fallback) and once after
    // create (the thread-local hit). free then repeats the fallback.
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);
    expect_argdescr_getters(nullptr, true);

    void* argdescr = nullptr;
    ASSERT_EQ(rocsparse_argdescr_create(&argdescr), rocsparse_status_success);
    ASSERT_NE(argdescr, nullptr);
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);
    expect_argdescr_getters(argdescr, true);
    expect_argdescr_getters(argdescr, false);

    EXPECT_EQ(rocsparse_argdescr_free(nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_free(argdescr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);
}

// Buffer-size has three direction combinations. Preprocess's two mixed
// diag-type launches are not on the itilu0 path.
TEST_F(HostInternalBranches, csxsldu_directions)
{
    constexpr auto row    = rocsparse_direction_row;
    constexpr auto column = rocsparse_direction_column;
    constexpr auto entry  = (4 + 1) * sizeof(rocsparse_int);

    const struct
    {
        rocsparse_direction dir, ldir, udir;
        size_t              bytes;
    } sizes[] = {{row, column, column, 2 * entry},
                 {row, row, row, 0},
                 {row, row, column, entry},
                 {column, row, row, 2 * entry},
                 {column, column, row, entry}};
    for(const auto& s : sizes)
    {
        size_t bytes = 99;
        EXPECT_EQ(sldu_buffer_size(handle, s.dir, s.ldir, s.udir, &bytes),
                  rocsparse_status_success);
        EXPECT_EQ(bytes, s.bytes);
    }

    device_vector<rocsparse_int> scratch{(size_t)8};
    ASSERT_TRUE(scratch.ptr);

    // 2x2 diagonal. Unit triangle stores nothing; non-unit stores the diagonal.
    // Row direction with a column factor writes the scanned pointers into scratch.
    auto expect_split = [&](rocsparse_diag_type        ldiag,
                            rocsparse_diag_type        udiag,
                            rocsparse_int              want_lnnz,
                            rocsparse_int              want_unnz,
                            std::vector<rocsparse_int> want) {
        ASSERT_EQ(hipMemset(scratch.ptr, 0x7f, 8 * sizeof(rocsparse_int)), hipSuccess);
        rocsparse_int lnnz = -1;
        rocsparse_int unnz = -1;
        EXPECT_EQ(
            sldu_preprocess(handle, row, ldiag, column, udiag, column, scratch.ptr, &lnnz, &unnz),
            rocsparse_status_success);
        EXPECT_EQ(lnnz, want_lnnz);
        EXPECT_EQ(unnz, want_unnz);
        ASSERT_EQ(to_host(scratch.ptr, want.size()), want);
    };
    expect_split(rocsparse_diag_type_unit, rocsparse_diag_type_non_unit, 0, 2, {0, 0, 0, 0, 1, 2});
    expect_split(rocsparse_diag_type_non_unit, rocsparse_diag_type_unit, 2, 0, {0, 1, 2, 0, 0, 0});

    // Factors in the input direction are written to lptr / uptr directly.
    std::vector<rocsparse_int> lptr, uptr;
    rocsparse_int              lnnz = -1, unnz = -1;
    EXPECT_EQ(sldu_preprocess(handle,
                              row,
                              rocsparse_diag_type_unit,
                              row,
                              rocsparse_diag_type_non_unit,
                              row,
                              scratch.ptr,
                              &lnnz,
                              &unnz,
                              &lptr,
                              &uptr),
              rocsparse_status_success);
    EXPECT_EQ(lnnz, 0);
    EXPECT_EQ(unnz, 2);
    EXPECT_EQ(lptr, (std::vector<rocsparse_int>{0, 0, 0}));
    EXPECT_EQ(uptr, (std::vector<rocsparse_int>{0, 1, 2}));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

// copy_csrmv_info's "already created" block and the device-pointer copies only
// run when the destination is live and the source actually owns arrays.
TEST_F(HostInternalBranches, copy_csrmv_info)
{
    EXPECT_EQ(rocsparse::copy_csrmv_info(nullptr, nullptr), rocsparse_status_invalid_pointer);

    _rocsparse_csrmv_info lone;
    EXPECT_EQ(rocsparse::copy_csrmv_info(&lone, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&lone, &lone), rocsparse_status_invalid_pointer);

    _rocsparse_csrmv_info empty_dest;
    _rocsparse_csrmv_info empty_src;
    EXPECT_EQ(rocsparse::copy_csrmv_info(&empty_dest, &empty_src), rocsparse_status_success);

    // n_rows_bins is copied as J_size * 32, which is 256 bytes for 64-bit indices.
    auto alloc = [](void** ptr) { ASSERT_EQ(hipMalloc(ptr, 256), hipSuccess); };

    auto fill
        = [&](_rocsparse_csrmv_info& info, rocsparse_indextype itype, rocsparse_indextype jtype) {
              info.m                  = 1;
              info.n                  = 1;
              info.nnz                = 1;
              info.max_rows           = 1;
              info.trans              = rocsparse_operation_transpose;
              info.adaptive.size      = 1;
              info.adaptive.first_row = 0;
              info.adaptive.last_row  = 1;
              info.lrb.size           = 1;
              info.index_type_I       = itype;
              info.index_type_J       = jtype;
              info.descr              = nullptr;
              info.csr_row_ptr        = nullptr;
              info.csr_col_ind        = nullptr;
              alloc(reinterpret_cast<void**>(&info.adaptive.row_blocks));
              alloc(reinterpret_cast<void**>(&info.adaptive.wg_flags));
              alloc(reinterpret_cast<void**>(&info.adaptive.wg_ids));
              alloc(reinterpret_cast<void**>(&info.lrb.wg_flags));
              alloc(&info.lrb.rows_offsets_scratch);
              alloc(&info.lrb.rows_bins);
              alloc(&info.lrb.n_rows_bins);
          };

    auto write = [](void* device, int32_t value, size_t offset = 0) {
        ASSERT_EQ(
            hipMemcpy(
                static_cast<char*>(device) + offset, &value, sizeof(value), hipMemcpyHostToDevice),
            hipSuccess);
    };
    auto read = [](const void* device, size_t offset = 0) {
        int32_t value = 0;
        EXPECT_EQ(hipMemcpy(&value,
                            static_cast<const char*>(device) + offset,
                            sizeof(value),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        return value;
    };
    // The device arrays of an info, each written with a distinct pattern.
    auto arrays = [](_rocsparse_csrmv_info& info) {
        return std::vector<void*>{info.adaptive.row_blocks,
                                  info.adaptive.wg_flags,
                                  info.adaptive.wg_ids,
                                  info.lrb.wg_flags,
                                  info.lrb.rows_offsets_scratch,
                                  info.lrb.rows_bins,
                                  info.lrb.n_rows_bins};
    };
    auto expect_scalars = [](const _rocsparse_csrmv_info& d, const _rocsparse_csrmv_info& s) {
        EXPECT_EQ(d.m, s.m);
        EXPECT_EQ(d.n, s.n);
        EXPECT_EQ(d.nnz, s.nnz);
        EXPECT_EQ(d.max_rows, s.max_rows);
        EXPECT_EQ(d.trans, s.trans);
        EXPECT_EQ(d.adaptive.size, s.adaptive.size);
        EXPECT_EQ(d.adaptive.first_row, s.adaptive.first_row);
        EXPECT_EQ(d.adaptive.last_row, s.adaptive.last_row);
        EXPECT_EQ(d.lrb.size, s.lrb.size);
        EXPECT_EQ(d.index_type_I, s.index_type_I);
        EXPECT_EQ(d.index_type_J, s.index_type_J);
    };

    _rocsparse_csrmv_info src;
    _rocsparse_csrmv_info dest;
    fill(src, rocsparse_indextype_i32, rocsparse_indextype_i32);
    const int32_t pattern = 0x13572468;
    for(size_t a = 0; a < arrays(src).size(); ++a)
    {
        write(arrays(src)[a], pattern + int32_t(a));
    }
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_success);
    expect_scalars(dest, src);
    for(size_t a = 0; a < arrays(dest).size(); ++a)
    {
        ASSERT_NE(arrays(dest)[a], nullptr) << a;
        EXPECT_NE(arrays(dest)[a], arrays(src)[a]) << a;
        EXPECT_EQ(read(arrays(dest)[a]), pattern + int32_t(a)) << a;
    }

    // A second copy reuses the destination's arrays.
    void* const   kept_blocks = dest.adaptive.row_blocks;
    const int32_t second      = 0x24681357;
    write(src.adaptive.row_blocks, second);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_success);
    EXPECT_EQ(dest.adaptive.row_blocks, kept_blocks);
    EXPECT_EQ(read(dest.adaptive.row_blocks), second);
    dest.m = 99;
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_invalid_pointer);

    _rocsparse_csrmv_info src64;
    _rocsparse_csrmv_info dest64;
    fill(src64, rocsparse_indextype_i64, rocsparse_indextype_i64);
    // The last 4 bytes of each 64-bit-sized copy: row_blocks is I_size * size,
    // n_rows_bins is J_size * 32.
    write(src64.adaptive.row_blocks, pattern, 4);
    write(src64.lrb.n_rows_bins, pattern, 252);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest64, &src64), rocsparse_status_success);
    expect_scalars(dest64, src64);
    ASSERT_NE(dest64.adaptive.row_blocks, nullptr);
    ASSERT_NE(dest64.lrb.n_rows_bins, nullptr);
    EXPECT_EQ(read(dest64.adaptive.row_blocks, 4), pattern);
    EXPECT_EQ(read(dest64.lrb.n_rows_bins, 252), pattern);
}

// Each debug switch is a full hit only when the value both changes and does not.
TEST_F(HostInternalBranches, debug_switches)
{
    struct Switch
    {
        void (*enable)();
        void (*disable)();
        int (*state)(); // The warnings switch has no query.
    };
    const Switch switches[]
        = {{rocsparse_enable_debug, rocsparse_disable_debug, rocsparse_state_debug},
           {rocsparse_enable_debug_verbose,
            rocsparse_disable_debug_verbose,
            rocsparse_state_debug_verbose},
           {rocsparse_enable_debug_warnings, rocsparse_disable_debug_warnings, nullptr},
           {rocsparse_enable_debug_arguments,
            rocsparse_disable_debug_arguments,
            rocsparse_state_debug_arguments},
           {rocsparse_enable_debug_arguments_verbose,
            rocsparse_disable_debug_arguments_verbose,
            rocsparse_state_debug_arguments_verbose},
           {rocsparse_enable_debug_force_host_assert,
            rocsparse_disable_debug_force_host_assert,
            rocsparse_state_debug_force_host_assert},
           {rocsparse_enable_debug_kernel_launch,
            rocsparse_disable_debug_kernel_launch,
            rocsparse_state_debug_kernel_launch}};

    const debug_state_guard restore;
    for(size_t i = 0; i < std::size(switches); ++i)
    {
        switches[i].enable();
        switches[i].enable();
        if(switches[i].state)
        {
            EXPECT_EQ(switches[i].state(), 1) << i;
        }
        switches[i].disable();
        switches[i].disable();
        if(switches[i].state)
        {
            EXPECT_EQ(switches[i].state(), 0) << i;
        }
    }
}
