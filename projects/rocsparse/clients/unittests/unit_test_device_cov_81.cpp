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

// Host paths still missed after the public clients and the earlier unit tests.
// Codecov counts a line only when every branch ran, so each check is taken on
// both sides. The CSR triangular split below does launch a tiny kernel; the
// handle and argument-descriptor checks do not.

#include "unit_test_utils.hpp"

#include <fstream>
#include <iterator>
#include <string>

#include "rocsparse-debugging.h"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_logging.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace rocsparse_ut;

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
    class EnvGuard
    {
    public:
        explicit EnvGuard(const char* name)
            : m_name(name)
        {
            const char* old = std::getenv(name);
            if(old != nullptr)
            {
                m_had   = true;
                m_saved = old;
            }
        }

        ~EnvGuard()
        {
            if(m_had)
            {
                setenv(m_name, m_saved.c_str(), 1);
            }
            else
            {
                unsetenv(m_name);
            }
        }

        EnvGuard(const EnvGuard&) = delete;
        EnvGuard& operator=(const EnvGuard&) = delete;

    private:
        const char* m_name;
        bool        m_had = false;
        std::string m_saved;
    };

    void set_arch_name(hipDeviceProp_t& prop, const char* name)
    {
        std::strncpy(prop.gcnArchName, name, sizeof(prop.gcnArchName) - 1);
        prop.gcnArchName[sizeof(prop.gcnArchName) - 1] = '\0';
    }

    rocsparse_status open_and_close(hipStream_t stream)
    {
        rocsparse_handle logged = nullptr;
        if(stream == nullptr)
        {
            rocsparse_status status = rocsparse_create_handle(&logged);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            return rocsparse_destroy_handle(logged);
        }
        rocsparse_status status = rocsparse_handle_create(&logged, stream, nullptr);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        return rocsparse_handle_destroy(logged, nullptr);
    }

    rocsparse_status sldu_buffer_size(rocsparse_handle    handle,
                                      rocsparse_direction dir,
                                      rocsparse_direction ldir,
                                      rocsparse_direction udir,
                                      size_t*             bytes)
    {
        return rocsparse::csxsldu_buffer_size_template<float, int32_t, int32_t>(
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

    rocsparse_status sldu_preprocess(rocsparse_handle    handle,
                                     rocsparse_direction dir,
                                     rocsparse_diag_type ldiag,
                                     rocsparse_direction ldir,
                                     rocsparse_diag_type udiag,
                                     rocsparse_direction udir,
                                     int32_t*            scratch,
                                     int32_t*            lnnz_out,
                                     int32_t*            unnz_out)
    {
        device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> ind{std::vector<int32_t>{0, 1}};
        device_vector<float>   val{std::vector<float>{1.0f, 1.0f}};
        device_vector<int32_t> lptr{(size_t)3};
        device_vector<int32_t> uptr{(size_t)3};
        if(!ptr.ptr || !ind.ptr || !val.ptr || !lptr.ptr || !uptr.ptr)
        {
            return rocsparse_status_memory_error;
        }
        int32_t                lnnz = 0;
        int32_t                unnz = 0;
        const rocsparse_status status
            = rocsparse::csxsldu_preprocess_template<float, int32_t, int32_t>(
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
        return status;
    }
} // namespace

class CovGap81 : public HandleTest
{
};

// Both handle constructors open trace, bench and debug logs only when
// ROCSPARSE_LAYER has those bits. The public clients leave the variable unset.
TEST_F(CovGap81, handle_layer_logs_and_arch)
{
    EnvGuard layer("ROCSPARSE_LAYER");
    EnvGuard trace("ROCSPARSE_LOG_TRACE_PATH");
    EnvGuard bench("ROCSPARSE_LOG_BENCH_PATH");
    EnvGuard debug("ROCSPARSE_LOG_DEBUG_PATH");
    setenv("ROCSPARSE_LAYER", "7", 1);
    unsetenv("ROCSPARSE_LOG_TRACE_PATH");
    unsetenv("ROCSPARSE_LOG_BENCH_PATH");
    unsetenv("ROCSPARSE_LOG_DEBUG_PATH");

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    // Paths unset: open_log_stream keeps std::cerr.
    EXPECT_EQ(open_and_close(nullptr), rocsparse_status_success);
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);

    setenv("ROCSPARSE_LOG_TRACE_PATH", "/tmp/rocsparse-cov81-trace.txt", 1);
    setenv("ROCSPARSE_LOG_BENCH_PATH", "/tmp/rocsparse-cov81-bench.txt", 1);
    setenv("ROCSPARSE_LOG_DEBUG_PATH", "/tmp/rocsparse-cov81-debug.txt", 1);

    rocsparse_handle logged = nullptr;
    ASSERT_EQ(rocsparse_create_handle(&logged), rocsparse_status_success);
    rocsparse::log_trace(nullptr, "cov81");
    rocsparse::log_trace(handle, "cov81");
    rocsparse::log_trace(logged, "cov81");
    rocsparse::log_trace(logged,
                         "cov81",
                         rocsparse_float_complex{1.0f, 2.0f},
                         rocsparse_double_complex{1.0, 2.0},
                         static_cast<_Float16>(1.0f),
                         rocsparse_bfloat16{1.0f});
    rocsparse::log_bench(nullptr, "cov81", std::string("s"));
    rocsparse::log_bench(logged, "cov81", std::string("s"), rocsparse_float_complex{1.0f, 2.0f});
    rocsparse::log_debug(nullptr, "cov81");
    rocsparse::log_debug(logged, "cov81");
    ASSERT_EQ(rocsparse_destroy_handle(logged), rocsparse_status_success);

    auto file_has = [](const char* path, const char* needle) {
        std::ifstream     in(path);
        const std::string text((std::istreambuf_iterator<char>(in)), {});
        return text.find(needle) != std::string::npos;
    };
    EXPECT_TRUE(file_has("/tmp/rocsparse-cov81-trace.txt", "cov81"));
    EXPECT_TRUE(file_has("/tmp/rocsparse-cov81-bench.txt", "cov81"));
    EXPECT_TRUE(file_has("/tmp/rocsparse-cov81-debug.txt", "cov81"));
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);

    // A missing directory leaves the ofstream closed, so the cerr fallback runs.
    setenv("ROCSPARSE_LOG_TRACE_PATH", "/no/such/dir/rocsparse-cov81-trace.txt", 1);
    setenv("ROCSPARSE_LOG_BENCH_PATH", "/no/such/dir/rocsparse-cov81-bench.txt", 1);
    setenv("ROCSPARSE_LOG_DEBUG_PATH", "/no/such/dir/rocsparse-cov81-debug.txt", 1);
    EXPECT_EQ(open_and_close(nullptr), rocsparse_status_success);
    EXPECT_EQ(open_and_close(stream), rocsparse_status_success);
    EXPECT_FALSE(std::ifstream("/no/such/dir/rocsparse-cov81-trace.txt").good());
    EXPECT_FALSE(std::ifstream("/no/such/dir/rocsparse-cov81-bench.txt").good());
    EXPECT_FALSE(std::ifstream("/no/such/dir/rocsparse-cov81-debug.txt").good());
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

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

// Every hipError_t arm, and every catch in the C++ exception translator.
TEST_F(CovGap81, hip_status_and_exceptions)
{
    const hipError_t       errors[]   = {hipSuccess,
                                 hipErrorMemoryAllocation,
                                 hipErrorLaunchOutOfResources,
                                 hipErrorInvalidDevicePointer,
                                 hipErrorInvalidDevice,
                                 hipErrorInvalidResourceHandle,
                                 hipErrorInvalidValue,
                                 hipErrorNoDevice,
                                 hipErrorUnknown,
                                 hipErrorNotReady};
    const rocsparse_status expected[] = {rocsparse_status_success,
                                         rocsparse_status_memory_error,
                                         rocsparse_status_memory_error,
                                         rocsparse_status_invalid_pointer,
                                         rocsparse_status_invalid_handle,
                                         rocsparse_status_invalid_handle,
                                         rocsparse_status_internal_error,
                                         rocsparse_status_internal_error,
                                         rocsparse_status_internal_error,
                                         rocsparse_status_internal_error};
    for(size_t i = 0; i < sizeof(errors) / sizeof(errors[0]); ++i)
    {
        EXPECT_EQ(rocsparse::get_rocsparse_status_for_hip_status(errors[i]), expected[i]);
    }

    EXPECT_EQ(rocsparse::exception_to_rocsparse_status(nullptr), rocsparse_status_success);
    try
    {
        throw rocsparse_status_invalid_pointer;
    }
    catch(...)
    {
        EXPECT_EQ(rocsparse::exception_to_rocsparse_status(std::current_exception()),
                  rocsparse_status_invalid_pointer);
    }
    try
    {
        throw std::bad_alloc();
    }
    catch(...)
    {
        EXPECT_EQ(rocsparse::exception_to_rocsparse_status(std::current_exception()),
                  rocsparse_status_memory_error);
    }
    try
    {
        throw 1;
    }
    catch(...)
    {
        EXPECT_EQ(rocsparse::exception_to_rocsparse_status(std::current_exception()),
                  rocsparse_status_thrown_exception);
    }
}

// The argument-descriptor getters are exported but the clients only read the
// name and the index. Verbose mode prints the descriptor that a bad argument
// just recorded.
TEST_F(CovGap81, argdescr_getters_and_verbose)
{
    const int was_verbose = rocsparse_state_debug_arguments_verbose();
    const int was_args    = rocsparse_state_debug_arguments();
    rocsparse_enable_debug_arguments();
    rocsparse_enable_debug_arguments_verbose();
    // Once before the descriptor exists (the stack fallback) and once after
    // create (the thread-local hit). free then repeats the fallback.
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_argdescr_get_msg(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_status(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_tid(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_pid(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_index(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_name(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_function_line(nullptr, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_function_name(nullptr, nullptr),
              rocsparse_status_invalid_pointer);

    void* argdescr = nullptr;
    ASSERT_EQ(rocsparse_argdescr_create(&argdescr), rocsparse_status_success);
    ASSERT_NE(argdescr, nullptr);
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_argdescr_get_msg(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_status(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_tid(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_pid(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_index(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_name(argdescr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_argdescr_get_function_line(argdescr, nullptr),
              rocsparse_status_invalid_pointer);

    const char*      msg           = nullptr;
    rocsparse_status status        = rocsparse_status_success;
    int64_t          tid           = -1;
    int64_t          pid           = -1;
    int              index         = -1;
    const char*      name          = nullptr;
    int              function_line = -1;
    const char*      function_name = nullptr;
    EXPECT_EQ(rocsparse_argdescr_get_msg(argdescr, &msg), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_status(argdescr, &status), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_tid(argdescr, &tid), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_pid(argdescr, &pid), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_index(argdescr, &index), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_name(argdescr, &name), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_function_line(argdescr, &function_line),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_get_function_name(argdescr, &function_name),
              rocsparse_status_success);
    EXPECT_NE(msg, nullptr);
    EXPECT_NE(name, nullptr);
    EXPECT_NE(function_name, nullptr);

    EXPECT_EQ(rocsparse_argdescr_free(nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_argdescr_free(argdescr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);

    if(was_verbose)
    {
        rocsparse_enable_debug_arguments_verbose();
    }
    else
    {
        rocsparse_disable_debug_arguments_verbose();
    }
    if(was_args)
    {
        rocsparse_enable_debug_arguments();
    }
    else
    {
        rocsparse_disable_debug_arguments();
    }
}

// Buffer-size has three direction combinations. Preprocess's column walk and
// the two mixed diag-type launches are not on the itilu0 path.
TEST_F(CovGap81, csxsldu_directions)
{
    size_t bytes = 99;
    EXPECT_EQ((sldu_buffer_size(handle,
                                rocsparse_direction_row,
                                rocsparse_direction_column,
                                rocsparse_direction_column,
                                &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, (4 + 1) * 2 * sizeof(int32_t));

    EXPECT_EQ((sldu_buffer_size(handle,
                                rocsparse_direction_row,
                                rocsparse_direction_row,
                                rocsparse_direction_row,
                                &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, size_t(0));

    EXPECT_EQ((sldu_buffer_size(handle,
                                rocsparse_direction_row,
                                rocsparse_direction_row,
                                rocsparse_direction_column,
                                &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, (4 + 1) * sizeof(int32_t));

    EXPECT_EQ((sldu_buffer_size(handle,
                                rocsparse_direction_column,
                                rocsparse_direction_row,
                                rocsparse_direction_row,
                                &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, (4 + 1) * 2 * sizeof(int32_t));

    EXPECT_EQ((sldu_buffer_size(handle,
                                rocsparse_direction_column,
                                rocsparse_direction_column,
                                rocsparse_direction_row,
                                &bytes)),
              rocsparse_status_success);
    EXPECT_EQ(bytes, (4 + 1) * sizeof(int32_t));

    device_vector<int32_t> scratch{(size_t)8};
    ASSERT_TRUE(scratch.ptr);

    // 2x2 diagonal. Unit triangle stores nothing; non-unit stores the diagonal.
    // Row direction with a column factor writes the scanned pointers into scratch.
    auto expect_split = [&](rocsparse_diag_type  ldiag,
                            rocsparse_diag_type  udiag,
                            int32_t              want_lnnz,
                            int32_t              want_unnz,
                            std::vector<int32_t> want) {
        ASSERT_EQ(hipMemset(scratch.ptr, 0x7f, 8 * sizeof(int32_t)), hipSuccess);
        int32_t lnnz = -1;
        int32_t unnz = -1;
        EXPECT_EQ((sldu_preprocess(handle,
                                   rocsparse_direction_row,
                                   ldiag,
                                   rocsparse_direction_column,
                                   udiag,
                                   rocsparse_direction_column,
                                   scratch.ptr,
                                   &lnnz,
                                   &unnz)),
                  rocsparse_status_success);
        EXPECT_EQ(lnnz, want_lnnz);
        EXPECT_EQ(unnz, want_unnz);
        const auto got = to_host(scratch.ptr, want.size());
        ASSERT_EQ(got, want);
    };
    expect_split(rocsparse_diag_type_unit, rocsparse_diag_type_non_unit, 0, 2, {0, 0, 0, 0, 1, 2});
    expect_split(rocsparse_diag_type_non_unit, rocsparse_diag_type_unit, 2, 0, {0, 1, 2, 0, 0, 0});
    auto*  raw          = reinterpret_cast<_rocsparse_handle*>(handle);
    size_t saved_buffer = raw->buffer_size;
    raw->buffer_size    = 0;
    EXPECT_EQ((sldu_preprocess(handle,
                               rocsparse_direction_column,
                               rocsparse_diag_type_unit,
                               rocsparse_direction_column,
                               rocsparse_diag_type_unit,
                               rocsparse_direction_column,
                               scratch.ptr,
                               nullptr,
                               nullptr)),
              rocsparse_status_success);
    raw->buffer_size = saved_buffer;
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

// copy_csrmv_info's "already created" block and the device-pointer copies only
// run when the destination is live and the source actually owns arrays.
TEST_F(CovGap81, copy_csrmv_info)
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

    _rocsparse_csrmv_info src;
    _rocsparse_csrmv_info dest;
    fill(src, rocsparse_indextype_i32, rocsparse_indextype_i32);
    int32_t pattern = 0x13572468;
    ASSERT_EQ(hipMemcpy(src.adaptive.row_blocks, &pattern, sizeof(pattern), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(src.lrb.n_rows_bins, &pattern, sizeof(pattern), hipMemcpyHostToDevice),
              hipSuccess);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_success);
    EXPECT_EQ(dest.m, src.m);
    EXPECT_EQ(dest.n, src.n);
    EXPECT_EQ(dest.nnz, src.nnz);
    EXPECT_EQ(dest.trans, src.trans);
    EXPECT_EQ(dest.adaptive.size, src.adaptive.size);
    EXPECT_EQ(dest.adaptive.first_row, src.adaptive.first_row);
    EXPECT_EQ(dest.adaptive.last_row, src.adaptive.last_row);
    EXPECT_NE(dest.adaptive.row_blocks, nullptr);
    EXPECT_NE(dest.adaptive.row_blocks, src.adaptive.row_blocks);
    int32_t copied = 0;
    ASSERT_EQ(hipMemcpy(&copied, dest.adaptive.row_blocks, sizeof(copied), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copied, pattern);
    ASSERT_EQ(hipMemcpy(&copied, dest.lrb.n_rows_bins, sizeof(copied), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copied, pattern);

    void* const kept_blocks = dest.adaptive.row_blocks;
    pattern                 = 0x24681357;
    ASSERT_EQ(hipMemcpy(src.adaptive.row_blocks, &pattern, sizeof(pattern), hipMemcpyHostToDevice),
              hipSuccess);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_success);
    EXPECT_EQ(dest.adaptive.row_blocks, kept_blocks);
    ASSERT_EQ(hipMemcpy(&copied, dest.adaptive.row_blocks, sizeof(copied), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(copied, pattern);
    dest.m = 99;
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest, &src), rocsparse_status_invalid_pointer);

    _rocsparse_csrmv_info src64;
    _rocsparse_csrmv_info dest64;
    fill(src64, rocsparse_indextype_i64, rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::copy_csrmv_info(&dest64, &src64), rocsparse_status_success);
}

// Each debug switch is a full hit only when the value both changes and does not.
TEST_F(CovGap81, debug_switches)
{
    const int debug    = rocsparse_state_debug();
    const int verbose  = rocsparse_state_debug_verbose();
    const int warnings = 0;
    const int force    = rocsparse_state_debug_force_host_assert();
    const int launch   = rocsparse_state_debug_kernel_launch();

    rocsparse_enable_debug();
    rocsparse_enable_debug();
    rocsparse_disable_debug();
    rocsparse_disable_debug();
    rocsparse_enable_debug_verbose();
    rocsparse_enable_debug_verbose();
    rocsparse_disable_debug_verbose();
    rocsparse_disable_debug_verbose();
    rocsparse_enable_debug_warnings();
    rocsparse_enable_debug_warnings();
    rocsparse_disable_debug_warnings();
    rocsparse_disable_debug_warnings();
    rocsparse_enable_debug_force_host_assert();
    rocsparse_enable_debug_force_host_assert();
    rocsparse_disable_debug_force_host_assert();
    rocsparse_disable_debug_force_host_assert();
    rocsparse_enable_debug_kernel_launch();
    rocsparse_enable_debug_kernel_launch();
    rocsparse_disable_debug_kernel_launch();
    rocsparse_disable_debug_kernel_launch();

    EXPECT_EQ(rocsparse_state_debug(), 0);
    EXPECT_EQ(rocsparse_state_debug_verbose(), 0);
    EXPECT_EQ(rocsparse_state_debug_force_host_assert(), 0);
    EXPECT_EQ(rocsparse_state_debug_kernel_launch(), 0);

    if(debug)
        rocsparse_enable_debug();
    if(verbose)
        rocsparse_enable_debug_verbose();
    if(warnings)
        rocsparse_enable_debug_warnings();
    if(force)
        rocsparse_enable_debug_force_host_assert();
    if(launch)
        rocsparse_enable_debug_kernel_launch();
}
